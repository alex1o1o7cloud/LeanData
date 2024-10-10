import Mathlib

namespace two_machines_total_copies_l678_67863

/-- Represents a copy machine with a constant copying rate -/
structure CopyMachine where
  rate : ℕ  -- copies per minute

/-- Calculates the total number of copies made by a machine in a given time -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Theorem: Two copy machines working together for 30 minutes will produce 3000 copies -/
theorem two_machines_total_copies 
  (machine1 : CopyMachine) 
  (machine2 : CopyMachine) 
  (h1 : machine1.rate = 35) 
  (h2 : machine2.rate = 65) : 
  copies_made machine1 30 + copies_made machine2 30 = 3000 := by
  sorry

#check two_machines_total_copies

end two_machines_total_copies_l678_67863


namespace johns_share_is_18_l678_67862

/-- The amount one person pays when splitting the cost of multiple items equally -/
def split_cost (num_items : ℕ) (price_per_item : ℚ) (num_people : ℕ) : ℚ :=
  (num_items : ℚ) * price_per_item / (num_people : ℚ)

/-- Theorem: John's share of the cake cost is $18 -/
theorem johns_share_is_18 :
  split_cost 3 12 2 = 18 := by
  sorry

end johns_share_is_18_l678_67862


namespace study_group_selection_probability_l678_67861

/-- Represents the probability of selecting a member with specific characteristics from a study group -/
def study_group_probability (women_percent : ℝ) (men_percent : ℝ) 
  (women_lawyer_percent : ℝ) (women_doctor_percent : ℝ) (women_engineer_percent : ℝ)
  (men_lawyer_percent : ℝ) (men_doctor_percent : ℝ) (men_engineer_percent : ℝ) : ℝ :=
  let woman_lawyer_prob := women_percent * women_lawyer_percent
  let man_doctor_prob := men_percent * men_doctor_percent
  woman_lawyer_prob + man_doctor_prob

/-- The probability of selecting a woman lawyer or a man doctor from the study group is 0.33 -/
theorem study_group_selection_probability : 
  study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30 = 0.33 := by
  sorry

#eval study_group_probability 0.65 0.35 0.40 0.30 0.30 0.50 0.20 0.30

end study_group_selection_probability_l678_67861


namespace vanilla_percentage_is_30_percent_l678_67898

def chocolate : ℕ := 70
def vanilla : ℕ := 90
def strawberry : ℕ := 50
def mint : ℕ := 30
def cookieDough : ℕ := 60

def totalResponses : ℕ := chocolate + vanilla + strawberry + mint + cookieDough

theorem vanilla_percentage_is_30_percent :
  (vanilla : ℚ) / (totalResponses : ℚ) * 100 = 30 := by
  sorry

end vanilla_percentage_is_30_percent_l678_67898


namespace derivative_of_exp_plus_x_l678_67840

open Real

theorem derivative_of_exp_plus_x (x : ℝ) :
  deriv (fun x => exp x + x) x = exp x + 1 := by sorry

end derivative_of_exp_plus_x_l678_67840


namespace sufficient_not_necessary_l678_67894

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The condition c < 0 is sufficient but not necessary for f(x) < 0 -/
theorem sufficient_not_necessary (b c : ℝ) :
  (c < 0 → ∃ x, f b c x < 0) ∧
  ∃ b' c' x', c' ≥ 0 ∧ f b' c' x' < 0 := by
  sorry

end sufficient_not_necessary_l678_67894


namespace unique_divisible_by_33_l678_67849

/-- Represents a five-digit number in the form 7n742 where n is a single digit -/
def number (n : ℕ) : ℕ := 70000 + n * 1000 + 742

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_33 :
  (isDivisibleBy (number 1) 33) ∧
  (∀ n : ℕ, n ≤ 9 → n ≠ 1 → ¬(isDivisibleBy (number n) 33)) :=
sorry

end unique_divisible_by_33_l678_67849


namespace new_students_admitted_l678_67885

theorem new_students_admitted (initial_students_per_section : ℕ) 
  (new_sections : ℕ) (final_total_sections : ℕ) (final_students_per_section : ℕ) :
  initial_students_per_section = 24 →
  new_sections = 3 →
  final_total_sections = 16 →
  final_students_per_section = 21 →
  (final_total_sections * final_students_per_section) - 
  ((final_total_sections - new_sections) * initial_students_per_section) = 24 := by
  sorry

end new_students_admitted_l678_67885


namespace product_of_three_numbers_l678_67820

theorem product_of_three_numbers (x y z n : ℝ) 
  (sum_eq : x + y + z = 180)
  (x_smallest : x ≤ y ∧ x ≤ z)
  (y_largest : y ≥ x ∧ y ≥ z)
  (n_def : n = 8 * x)
  (y_def : y = n + 10)
  (z_def : z = n - 10) :
  x * y * z = (180 / 17) * ((1440 / 17)^2 - 100) := by
  sorry

end product_of_three_numbers_l678_67820


namespace fraction_calculation_l678_67871

theorem fraction_calculation : (1/3 + 1/6) * 4/7 * 5/9 = 10/63 := by
  sorry

end fraction_calculation_l678_67871


namespace remainder_theorem_l678_67829

theorem remainder_theorem : (7 * 9^20 - 2^20) % 9 = 5 := by
  sorry

end remainder_theorem_l678_67829


namespace coefficient_x_cubed_expansion_l678_67809

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the polynomial expansion function
def expandPolynomial (a b : ℝ) (n : ℕ) : (ℕ → ℝ) := sorry

-- Theorem statement
theorem coefficient_x_cubed_expansion :
  let expansion := expandPolynomial 1 (-1) 5
  let coefficient_x_cubed := (expansion 3) + (expansion 1)
  coefficient_x_cubed = -15 := by sorry

end coefficient_x_cubed_expansion_l678_67809


namespace vector_on_line_l678_67896

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Two distinct vectors p and q define a line. The vector (3/5)*p + (2/5)*q lies on that line. -/
theorem vector_on_line (p q : V) (h : p ≠ q) :
  ∃ t : ℝ, (3/5 : ℝ) • p + (2/5 : ℝ) • q = p + t • (q - p) := by
  sorry

end vector_on_line_l678_67896


namespace cube_averaging_solution_l678_67869

/-- Represents a cube with real numbers on its vertices -/
structure Cube where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ
  H : ℝ

/-- Checks if the cube satisfies the averaging condition -/
def satisfiesAveraging (c : Cube) : Prop :=
  (c.D + c.E + c.B) / 3 = 6 ∧
  (c.A + c.F + c.C) / 3 = 3 ∧
  (c.D + c.G + c.B) / 3 = 6 ∧
  (c.A + c.C + c.H) / 3 = 4 ∧
  (c.A + c.H + c.F) / 3 = 3 ∧
  (c.E + c.G + c.B) / 3 = 6 ∧
  (c.H + c.F + c.C) / 3 = 5 ∧
  (c.D + c.G + c.E) / 3 = 3

/-- The theorem stating that the given solution is the only one satisfying the averaging condition -/
theorem cube_averaging_solution :
  ∀ c : Cube, satisfiesAveraging c →
    c.A = 0 ∧ c.B = 12 ∧ c.C = 6 ∧ c.D = 3 ∧ c.E = 3 ∧ c.F = 3 ∧ c.G = 3 ∧ c.H = 6 := by
  sorry

end cube_averaging_solution_l678_67869


namespace egg_game_probabilities_l678_67828

/-- Represents the color of an egg -/
inductive EggColor
| Yellow
| Red
| Blue

/-- Represents the game setup -/
structure EggGame where
  total_eggs : Nat
  yellow_eggs : Nat
  red_eggs : Nat
  blue_eggs : Nat
  game_fee : Int
  same_color_reward : Int
  diff_color_reward : Int

/-- Defines the game rules -/
def game : EggGame :=
  { total_eggs := 9
  , yellow_eggs := 3
  , red_eggs := 3
  , blue_eggs := 3
  , game_fee := 10
  , same_color_reward := 100
  , diff_color_reward := 10 }

/-- Event A: Picking a yellow egg on the first draw -/
def eventA (g : EggGame) : Rat :=
  g.yellow_eggs / g.total_eggs

/-- Event B: Winning the maximum reward -/
def eventB (g : EggGame) : Rat :=
  3 / Nat.choose g.total_eggs 3

/-- Probability of both events A and B occurring -/
def eventAB (g : EggGame) : Rat :=
  1 / Nat.choose g.total_eggs 3

/-- Expected profit from playing the game -/
def expectedProfit (g : EggGame) : Rat :=
  (g.same_color_reward - g.game_fee) * eventB g +
  (g.diff_color_reward - g.game_fee) * (9 / 28) +
  (-g.game_fee) * (18 / 28)

theorem egg_game_probabilities :
  eventA game = 1/3 ∧
  eventB game = 1/28 ∧
  eventAB game = eventA game * eventB game ∧
  expectedProfit game < 0 :=
sorry

end egg_game_probabilities_l678_67828


namespace solution_set_of_inequality_range_of_a_l678_67842

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ x + 5 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*a) → -5 ≤ a ∧ a ≤ 1 := by sorry

end solution_set_of_inequality_range_of_a_l678_67842


namespace division_of_eleven_by_five_l678_67841

theorem division_of_eleven_by_five :
  ∃ (A B : ℕ), 11 = 5 * A + B ∧ B < 5 ∧ A = 2 := by
  sorry

end division_of_eleven_by_five_l678_67841


namespace matches_played_l678_67827

/-- Represents the number of matches played -/
def n : ℕ := sorry

/-- The current batting average -/
def current_average : ℕ := 50

/-- The runs scored in the next match -/
def next_match_runs : ℕ := 78

/-- The new batting average after the next match -/
def new_average : ℕ := 54

/-- The total runs scored before the next match -/
def total_runs : ℕ := n * current_average

/-- The total runs after the next match -/
def new_total_runs : ℕ := total_runs + next_match_runs

/-- The theorem stating the number of matches played -/
theorem matches_played : n = 6 := by sorry

end matches_played_l678_67827


namespace max_rectangle_area_l678_67884

/-- The equation that the vertex coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- Theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
    ∀ (x' y' : ℝ), vertex_equation x' y' →
      rectangle_area x ≥ rectangle_area x' ∧
      rectangle_area x = 34.171875 :=
sorry

end max_rectangle_area_l678_67884


namespace ellipses_same_foci_l678_67812

/-- Given two ellipses with equations x²/9 + y²/4 = 1 and x²/(9-k) + y²/(4-k) = 1,
    where k < 4, prove that they have the same foci. -/
theorem ellipses_same_foci (k : ℝ) (h : k < 4) :
  let e1 := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  let e2 := {(x, y) : ℝ × ℝ | x^2 / (9 - k) + y^2 / (4 - k) = 1}
  let foci1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  let foci2 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 5 ∧ y = 0}
  foci1 = foci2 := by
sorry


end ellipses_same_foci_l678_67812


namespace cistern_filling_fraction_l678_67845

/-- Given a pipe that can fill a cistern in 55 minutes, 
    this theorem proves that the fraction of the cistern 
    filled in 5 minutes is 1/11. -/
theorem cistern_filling_fraction 
  (total_time : ℕ) 
  (filling_time : ℕ) 
  (h1 : total_time = 55) 
  (h2 : filling_time = 5) : 
  (filling_time : ℚ) / total_time = 1 / 11 :=
by sorry

end cistern_filling_fraction_l678_67845


namespace puppy_sale_revenue_l678_67832

/-- Calculates the total amount received from selling puppies --/
theorem puppy_sale_revenue (num_dogs : ℕ) (puppies_per_dog : ℕ) (sale_fraction : ℚ) (price_per_puppy : ℕ) : 
  num_dogs = 2 → 
  puppies_per_dog = 10 → 
  sale_fraction = 3/4 → 
  price_per_puppy = 200 → 
  (↑num_dogs * ↑puppies_per_dog : ℚ) * sale_fraction * ↑price_per_puppy = 3000 := by
  sorry

#check puppy_sale_revenue

end puppy_sale_revenue_l678_67832


namespace digit_1983_is_7_l678_67858

/-- Represents the decimal number formed by concatenating numbers from 1 to 999 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_1983_is_7 : nthDigit 1983 = 7 := by sorry

end digit_1983_is_7_l678_67858


namespace pyramid_stack_height_l678_67830

/-- Represents a stack of square blocks arranged in a stepped pyramid. -/
structure BlockStack where
  blockSideLength : ℝ
  numLayers : ℕ
  blocksPerLayer : ℕ → ℕ

/-- Calculates the total height of a block stack. -/
def totalHeight (stack : BlockStack) : ℝ :=
  stack.blockSideLength * stack.numLayers

/-- Theorem: The total height of a specific stepped pyramid stack is 30 cm. -/
theorem pyramid_stack_height :
  let stack : BlockStack := {
    blockSideLength := 10,
    numLayers := 3,
    blocksPerLayer := fun n => 3 - n + 1
  }
  totalHeight stack = 30 := by sorry

end pyramid_stack_height_l678_67830


namespace difference_between_x_and_y_l678_67838

theorem difference_between_x_and_y : 
  ∀ x y : ℤ, x = 10 ∧ y = 5 → x - y = 5 := by
  sorry

end difference_between_x_and_y_l678_67838


namespace largest_fraction_l678_67802

theorem largest_fraction : 
  (200 : ℚ) / 399 > 5 / 11 ∧
  (200 : ℚ) / 399 > 7 / 15 ∧
  (200 : ℚ) / 399 > 29 / 59 ∧
  (200 : ℚ) / 399 > 251 / 501 :=
by
  sorry

end largest_fraction_l678_67802


namespace circle_radius_equality_l678_67866

/-- The radius of a circle whose area is equal to the sum of the areas of four circles with radius 2 cm is 4 cm. -/
theorem circle_radius_equality (r : ℝ) : r > 0 → π * r^2 = 4 * (π * 2^2) → r = 4 := by
  sorry

end circle_radius_equality_l678_67866


namespace volume_of_cube_with_triple_surface_area_l678_67891

noncomputable def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

noncomputable def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area (v₁ : ℝ) (h₁ : v₁ = 64) :
  ∃ v₂ : ℝ, 
    (∃ s₁ s₂ : ℝ, 
      cube_volume s₁ = v₁ ∧ 
      cube_surface_area s₂ = 3 * cube_surface_area s₁ ∧ 
      cube_volume s₂ = v₂) ∧ 
    v₂ = 192 * Real.sqrt 3 := by
  sorry

end volume_of_cube_with_triple_surface_area_l678_67891


namespace two_wheeler_wheels_l678_67825

theorem two_wheeler_wheels (total_wheels : ℕ) (four_wheelers : ℕ) : total_wheels = 46 ∧ four_wheelers = 11 → 
  ∃ (two_wheelers : ℕ), two_wheelers * 2 + four_wheelers * 4 = total_wheels ∧ two_wheelers * 2 = 2 := by
  sorry

end two_wheeler_wheels_l678_67825


namespace canada_moose_population_l678_67800

/-- The moose population in Canada, in millions -/
def moose_population : ℝ := 1

/-- The beaver population in Canada, in millions -/
def beaver_population : ℝ := 2 * moose_population

/-- The human population in Canada, in millions -/
def human_population : ℝ := 38

theorem canada_moose_population :
  (beaver_population = 2 * moose_population) →
  (human_population = 19 * beaver_population) →
  (human_population = 38) →
  moose_population = 1 :=
by
  sorry

end canada_moose_population_l678_67800


namespace quadratic_root_theorem_l678_67819

/-- Represents an arithmetic sequence of three real numbers. -/
structure ArithmeticSequence (α : Type*) [LinearOrderedField α] where
  p : α
  q : α
  r : α
  is_arithmetic : q - r = p - q
  decreasing : p ≥ q ∧ q ≥ r
  nonnegative : r ≥ 0

/-- The theorem stating the properties of the quadratic equation and its root. -/
theorem quadratic_root_theorem (α : Type*) [LinearOrderedField α] 
  (seq : ArithmeticSequence α) : 
  (∃ x y : α, x = 2 * y ∧ 
   seq.p * x^2 + seq.q * x + seq.r = 0 ∧ 
   seq.p * y^2 + seq.q * y + seq.r = 0) → 
  (∃ y : α, y = -1/6 ∧ seq.p * y^2 + seq.q * y + seq.r = 0) :=
sorry

end quadratic_root_theorem_l678_67819


namespace calculation_proof_l678_67880

theorem calculation_proof : (12 * 0.5 * 3 * 0.0625 - 1.5) = -3/8 := by
  sorry

end calculation_proof_l678_67880


namespace arithmetic_operation_l678_67888

theorem arithmetic_operation : 3 * 14 + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end arithmetic_operation_l678_67888


namespace brown_eyed_brunettes_l678_67823

theorem brown_eyed_brunettes (total : ℕ) (blonde_blue : ℕ) (brunette : ℕ) (brown : ℕ)
  (h1 : total = 50)
  (h2 : blonde_blue = 14)
  (h3 : brunette = 31)
  (h4 : brown = 18) :
  brunette + blonde_blue - (total - brown) = 13 := by
  sorry

end brown_eyed_brunettes_l678_67823


namespace expression_equals_one_l678_67821

theorem expression_equals_one : 
  (50^2 - 9^2) / (40^2 - 8^2) * ((40 - 8) * (40 + 8)) / ((50 - 9) * (50 + 9)) = 1 := by
  sorry

end expression_equals_one_l678_67821


namespace min_rings_to_connect_five_links_l678_67882

/-- Represents a chain link with a specific number of rings -/
structure ChainLink where
  rings : ℕ

/-- Represents a collection of chain links -/
structure ChainCollection where
  links : List ChainLink

/-- Function to calculate the minimum number of rings to separate and reattach -/
def minRingsToConnect (chain : ChainCollection) : ℕ :=
  sorry

/-- Theorem stating the minimum number of rings to separate and reattach for the given problem -/
theorem min_rings_to_connect_five_links :
  let chain := ChainCollection.mk (List.replicate 5 (ChainLink.mk 3))
  minRingsToConnect chain = 3 := by
  sorry

end min_rings_to_connect_five_links_l678_67882


namespace base3_addition_proof_l678_67881

/-- Represents a single digit in base 3 -/
def Base3Digit := Fin 3

/-- Represents a three-digit number in base 3 -/
def Base3Number := Fin 27

def toBase3 (n : ℕ) : Base3Number :=
  Fin.ofNat (n % 27)

def fromBase3 (n : Base3Number) : ℕ :=
  n.val

def addBase3 (a b c : Base3Number) : Base3Number :=
  toBase3 (fromBase3 a + fromBase3 b + fromBase3 c)

theorem base3_addition_proof (C D : ℕ) 
  (h1 : C < 10 ∧ D < 10)
  (h2 : addBase3 (toBase3 (D * 10 + D)) (toBase3 (3 * 10 + 2)) (toBase3 (C * 100 + 2 * 10 + 4)) = 
        toBase3 (C * 100 + 2 * 10 + 4 + 1)) :
  toBase3 (if D > C then D - C else C - D) = toBase3 1 := by
  sorry

end base3_addition_proof_l678_67881


namespace absolute_value_inequality_l678_67893

theorem absolute_value_inequality (x : ℝ) :
  |x^2 - 5| < 9 ↔ -Real.sqrt 14 < x ∧ x < Real.sqrt 14 := by sorry

end absolute_value_inequality_l678_67893


namespace sams_cans_sams_final_can_count_l678_67856

/-- Sam's can collection problem -/
theorem sams_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) 
  (bags_given_away : ℕ) (large_cans_found : ℕ) : ℕ :=
  let total_bags := saturday_bags + sunday_bags
  let total_cans := total_bags * cans_per_bag
  let cans_given_away := bags_given_away * cans_per_bag
  let remaining_cans := total_cans - cans_given_away
  let large_cans_equivalent := large_cans_found * 2
  remaining_cans + large_cans_equivalent

/-- Proof of Sam's final can count -/
theorem sams_final_can_count : sams_cans 3 4 9 2 2 = 49 := by
  sorry

end sams_cans_sams_final_can_count_l678_67856


namespace angle_properties_l678_67824

theorem angle_properties (a : ℝ) (α : ℝ) (h1 : a < 0) 
  (h2 : Real.tan α = 2) 
  (h3 : (a, 2*a) ∈ Set.range (λ t : ℝ × ℝ => (t.1 * Real.cos α, t.1 * Real.sin α))) :
  Real.cos α = -Real.sqrt 5 / 5 ∧ 
  Real.tan α = 2 ∧ 
  (Real.cos α)^2 / Real.tan α = 1/10 := by
sorry

end angle_properties_l678_67824


namespace mother_pies_per_day_l678_67889

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The number of pies Eddie can bake per day -/
def eddie_pies : ℕ := 3

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- The total number of pies they can bake in the given days -/
def total_pies : ℕ := 119

theorem mother_pies_per_day :
  eddie_pies * days + sister_pies * days + mother_pies * days = total_pies :=
by sorry

end mother_pies_per_day_l678_67889


namespace smaller_number_is_ten_l678_67852

theorem smaller_number_is_ten (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) (h3 : x ≤ y) : x = 10 := by
  sorry

end smaller_number_is_ten_l678_67852


namespace nicolai_ate_six_pounds_l678_67839

/-- The amount of fruit eaten by Nicolai given the total fruit eaten and the amounts eaten by Mario and Lydia. -/
def nicolai_fruit (total_fruit : ℚ) (mario_fruit : ℚ) (lydia_fruit : ℚ) : ℚ :=
  total_fruit - (mario_fruit + lydia_fruit)

/-- Converts ounces to pounds -/
def ounces_to_pounds (ounces : ℚ) : ℚ :=
  ounces / 16

theorem nicolai_ate_six_pounds 
  (total_fruit : ℚ)
  (mario_ounces : ℚ)
  (lydia_ounces : ℚ)
  (h_total : total_fruit = 8)
  (h_mario : mario_ounces = 8)
  (h_lydia : lydia_ounces = 24) :
  nicolai_fruit total_fruit (ounces_to_pounds mario_ounces) (ounces_to_pounds lydia_ounces) = 6 := by
  sorry

#check nicolai_ate_six_pounds

end nicolai_ate_six_pounds_l678_67839


namespace greatest_multiple_of_8_remainder_l678_67873

def is_valid_number (n : ℕ) : Prop :=
  ∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ 0 ∧ d₂ ≠ 0

theorem greatest_multiple_of_8_remainder (M : ℕ) : 
  (∀ n, n > M → ¬(is_valid_number n ∧ 8 ∣ n)) →
  is_valid_number M →
  8 ∣ M →
  M % 1000 = 984 :=
sorry

end greatest_multiple_of_8_remainder_l678_67873


namespace equilateral_triangles_count_l678_67836

/-- The number of points evenly spaced on a circle -/
def n : ℕ := 900

/-- The number of points needed to form an equilateral triangle on the circle -/
def equilateral_spacing : ℕ := n / 3

/-- The number of equilateral triangles with all vertices on the circle -/
def all_vertices_on_circle : ℕ := equilateral_spacing

/-- The number of ways to choose 2 points from n points -/
def choose_two : ℕ := n * (n - 1) / 2

/-- The total number of equilateral triangles with at least two vertices from the n points -/
def total_triangles : ℕ := 2 * choose_two - all_vertices_on_circle

theorem equilateral_triangles_count : total_triangles = 808800 := by
  sorry

end equilateral_triangles_count_l678_67836


namespace power_three_minus_two_plus_three_l678_67877

theorem power_three_minus_two_plus_three : 2^3 - 2 + 3 = 9 := by
  sorry

end power_three_minus_two_plus_three_l678_67877


namespace angle_triple_supplement_l678_67848

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end angle_triple_supplement_l678_67848


namespace train_length_l678_67851

/-- The length of a train given its crossing times over a bridge and a lamp post -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ)
  (h1 : bridge_length = 150)
  (h2 : bridge_time = 7.5)
  (h3 : lamp_time = 2.5)
  (h4 : bridge_time > 0)
  (h5 : lamp_time > 0) :
  ∃ (train_length : ℝ),
    train_length = 75 ∧
    (train_length + bridge_length) / bridge_time = train_length / lamp_time :=
by sorry

end train_length_l678_67851


namespace sum_of_squares_of_roots_l678_67807

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 6 * a^2 + 9 * a + 18 = 0) →
  (3 * b^3 - 6 * b^2 + 9 * b + 18 = 0) →
  (3 * c^3 - 6 * c^2 + 9 * c + 18 = 0) →
  a^2 + b^2 + c^2 = -2 := by
sorry

end sum_of_squares_of_roots_l678_67807


namespace second_period_odds_correct_l678_67857

/-- Represents the types of light bulbs -/
inductive BulbType
  | A
  | B
  | C

/-- Initial odds of burning out for each bulb type -/
def initialOdds (t : BulbType) : ℚ :=
  match t with
  | BulbType.A => 1/3
  | BulbType.B => 1/4
  | BulbType.C => 1/5

/-- Decrease rate for each bulb type -/
def decreaseRate (t : BulbType) : ℚ :=
  match t with
  | BulbType.A => 1/2
  | BulbType.B => 1/3
  | BulbType.C => 1/4

/-- Calculates the odds of burning out for the second 6-month period -/
def secondPeriodOdds (t : BulbType) : ℚ :=
  initialOdds t * decreaseRate t

/-- Theorem stating the odds of burning out for each bulb type in the second 6-month period -/
theorem second_period_odds_correct :
  (secondPeriodOdds BulbType.A = 1/6) ∧
  (secondPeriodOdds BulbType.B = 1/12) ∧
  (secondPeriodOdds BulbType.C = 1/20) := by
  sorry

end second_period_odds_correct_l678_67857


namespace complement_N_subset_complement_M_l678_67814

/-- The set of real numbers -/
def R : Set ℝ := Set.univ

/-- The set M defined as {x | 0 < x < 2} -/
def M : Set ℝ := {x | 0 < x ∧ x < 2}

/-- The set N defined as {x | x^2 + x - 6 ≤ 0} -/
def N : Set ℝ := {x | x^2 + x - 6 ≤ 0}

/-- Theorem stating that the complement of N is a subset of the complement of M -/
theorem complement_N_subset_complement_M : (R \ N) ⊆ (R \ M) := by
  sorry

end complement_N_subset_complement_M_l678_67814


namespace johnny_guitar_picks_l678_67813

theorem johnny_guitar_picks (total red blue yellow : ℕ) : 
  total > 0 → 
  2 * red = total → 
  3 * blue = total → 
  yellow = total - red - blue → 
  blue = 12 → 
  yellow = 6 := by
sorry

end johnny_guitar_picks_l678_67813


namespace furniture_cost_price_sum_l678_67818

theorem furniture_cost_price_sum (sp1 sp2 sp3 sp4 : ℕ) 
  (h1 : sp1 = 3000) (h2 : sp2 = 2400) (h3 : sp3 = 12000) (h4 : sp4 = 18000) : 
  (sp1 / 120 * 100 + sp2 / 120 * 100 + sp3 / 120 * 100 + sp4 / 120 * 100 : ℕ) = 29500 := by
  sorry

#check furniture_cost_price_sum

end furniture_cost_price_sum_l678_67818


namespace cube_painting_l678_67835

/-- Given a cube of side length n constructed from n³ smaller cubes,
    if (n-2)³ = 343 small cubes remain unpainted after some faces are painted,
    then exactly 3 faces of the large cube must have been painted. -/
theorem cube_painting (n : ℕ) (h : (n - 2)^3 = 343) :
  ∃ (painted_faces : ℕ), painted_faces = 3 ∧ painted_faces < 6 := by
  sorry

end cube_painting_l678_67835


namespace seven_eighths_of_sixteen_thirds_l678_67844

theorem seven_eighths_of_sixteen_thirds :
  (7 / 8 : ℚ) * (16 / 3 : ℚ) = 14 / 3 := by
  sorry

end seven_eighths_of_sixteen_thirds_l678_67844


namespace function_properties_l678_67872

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log (x + b)) / x

noncomputable def g (a x : ℝ) : ℝ := x + 2 / x - a - 2

noncomputable def F (a b x : ℝ) : ℝ := f a b x + g a x

theorem function_properties (a b : ℝ) (ha : a ≤ 2) (ha_nonzero : a ≠ 0) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = f a b x) →
  (∃ m : ℝ, ∀ x : ℝ, f a b x - f a b 1 = m * (x - 1) → f a b 3 = 0) →
  (∃! x : ℝ, x ∈ Set.Ioo 0 2 ∧ F a b x = 0) →
  (b = 2 * a ∧ (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2))) :=
sorry

end function_properties_l678_67872


namespace equation_solutions_l678_67833

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ 
    x₁^2 - 2*x₁ - 2 = 0 ∧ x₂^2 - 2*x₂ - 2 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3/2 ∧ y₂ = 7/2 ∧ 
    2*(y₁ - 3)^2 = y₁ - 3 ∧ 2*(y₂ - 3)^2 = y₂ - 3) :=
by sorry

end equation_solutions_l678_67833


namespace stratified_sample_size_l678_67834

/-- Represents the quantity ratio of products A, B, and C -/
def quantity_ratio : Fin 3 → ℕ
  | 0 => 2  -- Product A
  | 1 => 3  -- Product B
  | 2 => 5  -- Product C
  | _ => 0  -- Unreachable case

/-- The sample size of product A -/
def sample_size_A : ℕ := 10

/-- Calculates the total sample size based on the sample size of product A -/
def total_sample_size (sample_A : ℕ) : ℕ :=
  sample_A * (quantity_ratio 0 + quantity_ratio 1 + quantity_ratio 2) / quantity_ratio 0

theorem stratified_sample_size :
  total_sample_size sample_size_A = 50 := by
  sorry

end stratified_sample_size_l678_67834


namespace meeting_equation_correct_l678_67874

/-- Represents the scenario of two people meeting on a straight road -/
def meeting_equation (x : ℝ) : Prop :=
  x / 6 + (x - 1) / 4 = 1

/-- The time it takes for A to travel the entire distance -/
def time_A : ℝ := 4

/-- The time it takes for B to travel the entire distance -/
def time_B : ℝ := 6

/-- The time difference between A and B starting their journey -/
def time_difference : ℝ := 1

/-- Theorem stating that the meeting equation correctly represents the scenario -/
theorem meeting_equation_correct (x : ℝ) :
  (x ≥ time_difference) →
  (x / time_B + (x - time_difference) / time_A = 1) ↔ meeting_equation x :=
by sorry

end meeting_equation_correct_l678_67874


namespace investment_duration_l678_67843

/-- Represents a partner in the investment scenario -/
structure Partner where
  investment : ℚ
  profit : ℚ
  duration : ℚ

/-- The investment scenario with two partners -/
def InvestmentScenario (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 10 ∧
  p.duration = 8

theorem investment_duration (p q : Partner) 
  (h : InvestmentScenario p q) : q.duration = 16 := by
  sorry

end investment_duration_l678_67843


namespace quadratic_equation_properties_l678_67837

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m+2)*x + m
  -- The equation always has two distinct real roots
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  -- When the sum condition is satisfied, m = 3
  (x₁ + x₂ + 2*x₁*x₂ = 1 → m = 3) :=
by sorry

end quadratic_equation_properties_l678_67837


namespace average_temperature_problem_l678_67822

theorem average_temperature_problem (T₁ T₂ T₃ T₄ T₅ : ℚ) : 
  (T₁ + T₂ + T₃ + T₄) / 4 = 58 →
  T₁ / T₅ = 7 / 8 →
  T₅ = 32 →
  (T₂ + T₃ + T₄ + T₅) / 4 = 59 :=
by sorry

end average_temperature_problem_l678_67822


namespace quadratic_roots_relation_l678_67855

/-- Given two quadratic functions f and g, if f has two distinct real roots,
    then g must have at least one real root. -/
theorem quadratic_roots_relation (a b c : ℝ) (h : a * c ≠ 0) :
  let f := fun x : ℝ ↦ a * x^2 + b * x + c
  let g := fun x : ℝ ↦ c * x^2 + b * x + a
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ z : ℝ, g z = 0) :=
by
  sorry

end quadratic_roots_relation_l678_67855


namespace quadratic_sum_l678_67806

/-- Given a quadratic polynomial 12x^2 + 144x + 1728, when written in the form a(x+b)^2+c
    where a, b, and c are constants, prove that a + b + c = 1314 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (12 * x^2 + 144 * x + 1728 = a * (x + b)^2 + c) ∧ (a + b + c = 1314) := by
  sorry

end quadratic_sum_l678_67806


namespace boys_without_calculators_l678_67865

theorem boys_without_calculators (total_boys : ℕ) (total_with_calculators : ℕ) (girls_with_calculators : ℕ) : 
  total_boys = 20 →
  total_with_calculators = 30 →
  girls_with_calculators = 18 →
  total_boys - (total_with_calculators - girls_with_calculators) = 8 :=
by sorry

end boys_without_calculators_l678_67865


namespace marble_probability_l678_67879

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 20) (h2 : blue = 5) (h3 : red = 7) :
  let white := total - (blue + red)
  (red + white : ℚ) / total = 3 / 4 := by
sorry

end marble_probability_l678_67879


namespace checkerboard_coverage_l678_67815

/-- Represents a checkerboard --/
structure Checkerboard where
  rows : ℕ
  cols : ℕ
  removed_squares : ℕ

/-- Checks if a checkerboard can be covered by dominoes --/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem stating which boards can be covered --/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered board ↔ 
  (board ≠ ⟨4, 4, 1⟩ ∧ board ≠ ⟨3, 7, 0⟩ ∧ board ≠ ⟨7, 3, 0⟩) :=
sorry

end checkerboard_coverage_l678_67815


namespace max_median_length_l678_67817

theorem max_median_length (a b c m : ℝ) (hA : Real.cos A = 15/17) (ha : a = 2) :
  m ≤ 4 ∧ ∃ (b c : ℝ), m = 4 := by
  sorry

end max_median_length_l678_67817


namespace fifth_term_is_five_l678_67850

def fibonacci_like_sequence : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci_like_sequence (n + 1) + fibonacci_like_sequence n

theorem fifth_term_is_five : fibonacci_like_sequence 4 = 5 := by
  sorry

end fifth_term_is_five_l678_67850


namespace ahmed_orange_trees_count_l678_67801

-- Define the number of apple and orange trees for Hassan
def hassan_apple_trees : ℕ := 1
def hassan_orange_trees : ℕ := 2

-- Define the number of apple trees for Ahmed
def ahmed_apple_trees : ℕ := 4 * hassan_apple_trees

-- Define the total number of trees for Hassan
def hassan_total_trees : ℕ := hassan_apple_trees + hassan_orange_trees

-- Define the relationship between Ahmed's and Hassan's total trees
def ahmed_total_trees (ahmed_orange_trees : ℕ) : ℕ := 
  ahmed_apple_trees + ahmed_orange_trees

-- Theorem stating that Ahmed has 8 orange trees
theorem ahmed_orange_trees_count : 
  ∃ (x : ℕ), ahmed_total_trees x = hassan_total_trees + 9 ∧ x = 8 := by
  sorry

end ahmed_orange_trees_count_l678_67801


namespace line_intersection_with_circle_l678_67847

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define a line with slope 1
def line_with_slope_1 (x y m : ℝ) : Prop := y = x + m

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define a chord of the circle
def chord (A B : ℝ × ℝ) : Prop :=
  point_on_circle A.1 A.2 ∧ point_on_circle B.1 B.2

-- Define a circle passing through three points
def circle_through_points (A B O : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (O.1 - center.1)^2 + (O.2 - center.2)^2 = radius^2

-- Theorem statement
theorem line_intersection_with_circle :
  ∃ (m : ℝ), m = 1 ∨ m = -4 ∧
  ∀ (x y : ℝ),
    line_with_slope_1 x y m →
    (∃ (A B : ℝ × ℝ),
      chord A B ∧
      line_with_slope_1 A.1 A.2 m ∧
      line_with_slope_1 B.1 B.2 m ∧
      circle_through_points A B origin) :=
by sorry

end line_intersection_with_circle_l678_67847


namespace product_and_difference_equation_l678_67804

theorem product_and_difference_equation (n v : ℝ) : 
  n = -4.5 → 10 * n = v - 2 * n → v = -9 := by sorry

end product_and_difference_equation_l678_67804


namespace frog_jump_distance_l678_67875

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_grasshopper_diff : ℕ) 
  (h1 : grasshopper_jump = 36)
  (h2 : frog_grasshopper_diff = 17) :
  grasshopper_jump + frog_grasshopper_diff = 53 :=
by sorry

end frog_jump_distance_l678_67875


namespace equation_solution_l678_67803

theorem equation_solution : 
  ∃ x : ℝ, (5 + 3.5 * x = 2 * x - 25 + x) ∧ (x = -60) := by
  sorry

end equation_solution_l678_67803


namespace equation_solution_l678_67810

theorem equation_solution (x : ℝ) : 
  x = 46 →
  (8 / (Real.sqrt (x - 10) - 10) + 
   2 / (Real.sqrt (x - 10) - 5) + 
   9 / (Real.sqrt (x - 10) + 5) + 
   15 / (Real.sqrt (x - 10) + 10) = 0) :=
by sorry

end equation_solution_l678_67810


namespace sum_of_x_and_y_is_two_l678_67860

theorem sum_of_x_and_y_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997 * (x - 1) = -1)
  (hy : (y - 1)^3 + 1997 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end sum_of_x_and_y_is_two_l678_67860


namespace james_total_points_l678_67886

/-- Represents the quiz bowl game rules and James' performance --/
structure QuizBowl where
  points_per_correct : ℕ := 2
  points_per_incorrect : ℕ := 1
  bonus_points : ℕ := 4
  total_rounds : ℕ := 5
  questions_per_round : ℕ := 5
  james_correct_answers : ℕ := 24
  james_unanswered : ℕ := 1

/-- Calculates the total points James earned in the quiz bowl --/
def calculate_points (game : QuizBowl) : ℕ :=
  let total_questions := game.total_rounds * game.questions_per_round
  let points_from_correct := game.james_correct_answers * game.points_per_correct
  let full_rounds := (total_questions - game.james_unanswered - game.james_correct_answers) / game.questions_per_round
  let bonus_points := full_rounds * game.bonus_points
  points_from_correct + bonus_points

/-- Theorem stating that James' total points in the quiz bowl are 64 --/
theorem james_total_points (game : QuizBowl) : calculate_points game = 64 := by
  sorry

end james_total_points_l678_67886


namespace coin_split_sum_l678_67899

/-- Represents the sum of recorded products when splitting coins into piles -/
def recordedSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 25 coins, the sum of recorded products is 300 -/
theorem coin_split_sum :
  recordedSum 25 = 300 := by
  sorry

end coin_split_sum_l678_67899


namespace range_of_m_l678_67846

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x = x^2 - 4*x - 6) →
  (Set.range f = Set.Icc (-10) (-6)) →
  m ∈ Set.Icc 2 4 :=
sorry

end range_of_m_l678_67846


namespace threes_squared_threes_2009_squared_l678_67887

/-- Represents a number consisting of n repeated digits -/
def repeated_digit (d : Nat) (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | k + 1 => d + 10 * (repeated_digit d k)

/-- The theorem to be proved -/
theorem threes_squared (n : Nat) (h : n > 0) :
  (repeated_digit 3 n) ^ 2 = 
    repeated_digit 1 (n-1) * 10^n + 
    repeated_digit 8 (n-1) * 10 + 9 := by
  sorry

/-- The specific case for 2009 threes -/
theorem threes_2009_squared :
  (repeated_digit 3 2009) ^ 2 = 
    repeated_digit 1 2008 * 10^2009 + 
    repeated_digit 8 2008 * 10 + 9 := by
  sorry

end threes_squared_threes_2009_squared_l678_67887


namespace inscribed_circle_radius_l678_67811

theorem inscribed_circle_radius (A₁ A₂ : ℝ) (h1 : A₁ > 0) (h2 : A₂ > 0) : 
  (A₁ + A₂ = π * 8^2) →
  (A₂ = (A₁ + (A₁ + A₂)) / 2) →
  A₁ = π * ((8 * Real.sqrt 3) / 3)^2 :=
by sorry

end inscribed_circle_radius_l678_67811


namespace twenty_percent_of_three_and_three_quarters_l678_67816

theorem twenty_percent_of_three_and_three_quarters :
  (20 : ℚ) / 100 * (15 : ℚ) / 4 = (3 : ℚ) / 4 := by sorry

end twenty_percent_of_three_and_three_quarters_l678_67816


namespace new_xanadu_license_plates_l678_67876

/-- The number of possible letters in each letter position of a license plate. -/
def num_letters : ℕ := 26

/-- The number of possible digits in each digit position of a license plate. -/
def num_digits : ℕ := 10

/-- The total number of valid license plates in New Xanadu. -/
def total_license_plates : ℕ := num_letters ^ 3 * num_digits ^ 3

/-- Theorem stating the total number of valid license plates in New Xanadu. -/
theorem new_xanadu_license_plates : total_license_plates = 17576000 := by
  sorry

end new_xanadu_license_plates_l678_67876


namespace complex_function_evaluation_l678_67808

theorem complex_function_evaluation : 
  let z : ℂ := (Complex.I + 1) / (Complex.I - 1)
  let f : ℂ → ℂ := fun x ↦ x^2 - x + 1
  f z = Complex.I := by sorry

end complex_function_evaluation_l678_67808


namespace gross_monthly_salary_l678_67854

theorem gross_monthly_salary (rent food_expenses mortgage savings taxes gross_salary : ℚ) : 
  rent = 600 →
  food_expenses = (3/5) * rent →
  mortgage = 3 * food_expenses →
  savings = 2000 →
  taxes = (2/5) * savings →
  gross_salary = rent + food_expenses + mortgage + taxes + savings →
  gross_salary = 4840 := by
sorry

end gross_monthly_salary_l678_67854


namespace black_white_ratio_after_border_l678_67805

/-- Represents a rectangular tile pattern -/
structure TilePattern where
  length : ℕ
  width : ℕ
  blackTiles : ℕ
  whiteTiles : ℕ

/-- Adds a border of black tiles to a tile pattern -/
def addBorder (pattern : TilePattern) (borderWidth : ℕ) : TilePattern :=
  { length := pattern.length + 2 * borderWidth,
    width := pattern.width + 2 * borderWidth,
    blackTiles := pattern.blackTiles + 
      (pattern.length + pattern.width + 2 * borderWidth) * 2 * borderWidth + 4 * borderWidth^2,
    whiteTiles := pattern.whiteTiles }

theorem black_white_ratio_after_border (initialPattern : TilePattern) :
  initialPattern.length = 4 →
  initialPattern.width = 8 →
  initialPattern.blackTiles = 10 →
  initialPattern.whiteTiles = 22 →
  let finalPattern := addBorder initialPattern 2
  (finalPattern.blackTiles : ℚ) / finalPattern.whiteTiles = 19 / 11 := by
  sorry

end black_white_ratio_after_border_l678_67805


namespace yeast_population_after_30_minutes_l678_67892

/-- The population of yeast cells after a given time period. -/
def yeast_population (initial_population : ℕ) (time_minutes : ℕ) : ℕ :=
  initial_population * (3 ^ (time_minutes / 5))

/-- Theorem: The yeast population after 30 minutes is 36450 cells. -/
theorem yeast_population_after_30_minutes :
  yeast_population 50 30 = 36450 := by
  sorry

end yeast_population_after_30_minutes_l678_67892


namespace distance_between_x_and_y_l678_67897

-- Define the walking speeds
def yolanda_speed : ℝ := 2
def bob_speed : ℝ := 4

-- Define the time difference in starting
def time_difference : ℝ := 1

-- Define Bob's distance walked when they meet
def bob_distance : ℝ := 25.333333333333332

-- Define the total distance between X and Y
def total_distance : ℝ := 40

-- Theorem statement
theorem distance_between_x_and_y :
  let time_bob_walked := bob_distance / bob_speed
  let yolanda_distance := yolanda_speed * (time_bob_walked + time_difference)
  yolanda_distance + bob_distance = total_distance := by
  sorry

end distance_between_x_and_y_l678_67897


namespace exponential_characterization_l678_67868

def is_exponential (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

theorem exponential_characterization (f : ℝ → ℝ) 
  (h1 : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :
  is_exponential f :=
sorry

end exponential_characterization_l678_67868


namespace max_value_sin_cos_l678_67878

theorem max_value_sin_cos (θ : Real) (h : 0 < θ ∧ θ < Real.pi) :
  (∀ φ, 0 < φ ∧ φ < Real.pi → 
    Real.sin (φ / 2) * (1 + Real.cos φ) ≤ Real.sin (θ / 2) * (1 + Real.cos θ)) ↔ 
  Real.sin (θ / 2) * (1 + Real.cos θ) = 4 * Real.sqrt 3 / 9 :=
by sorry

end max_value_sin_cos_l678_67878


namespace student_pet_difference_l678_67883

/-- Represents a fourth-grade classroom at Maplewood School -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  guinea_pigs : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A standard fourth-grade classroom at Maplewood School -/
def standard_classroom : Classroom :=
  { students := 24
  , rabbits := 3
  , guinea_pigs := 2 }

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * standard_classroom.students

/-- The total number of pets (rabbits and guinea pigs) in all classrooms -/
def total_pets : ℕ := num_classrooms * (standard_classroom.rabbits + standard_classroom.guinea_pigs)

/-- Theorem: The difference between the total number of students and the total number of pets is 95 -/
theorem student_pet_difference : total_students - total_pets = 95 := by
  sorry

end student_pet_difference_l678_67883


namespace pants_price_l678_67853

theorem pants_price (total coat pants shoes : ℕ) 
  (h1 : total = 700)
  (h2 : total = coat + pants + shoes)
  (h3 : coat = pants + 340)
  (h4 : coat = shoes + pants + 180) :
  pants = 100 := by
  sorry

end pants_price_l678_67853


namespace white_squares_37th_row_l678_67864

/-- Represents the number of squares in a row of the stair-step figure -/
def num_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def num_white_squares (n : ℕ) : ℕ := (num_squares n + 1) / 2

theorem white_squares_37th_row :
  num_white_squares 37 = 37 := by sorry

end white_squares_37th_row_l678_67864


namespace rotational_cipher_key_l678_67890

/-- Represents the encoding function for a rotational cipher --/
def encode (key : ℕ) (letter : ℕ) : ℕ :=
  ((letter + key - 1) % 26) + 1

/-- Theorem: If the sum of encoded values for A, B, and C is 52, the key is 25 --/
theorem rotational_cipher_key (key : ℕ) 
  (h1 : 1 ≤ key ∧ key ≤ 26) 
  (h2 : encode key 1 + encode key 2 + encode key 3 = 52) : 
  key = 25 := by
  sorry

#check rotational_cipher_key

end rotational_cipher_key_l678_67890


namespace pie_chart_probability_l678_67826

theorem pie_chart_probability (pE pF pG pH : ℚ) : 
  pE = 1/3 →
  pF = 1/6 →
  pG = pH →
  pE + pF + pG + pH = 1 →
  pG = 1/4 := by
sorry

end pie_chart_probability_l678_67826


namespace binomial_13_11_times_2_l678_67859

theorem binomial_13_11_times_2 : 2 * Nat.choose 13 11 = 156 := by
  sorry

end binomial_13_11_times_2_l678_67859


namespace nobel_laureates_count_l678_67867

/-- The number of Nobel Prize laureates at a workshop given specific conditions -/
theorem nobel_laureates_count (total : ℕ) (wolf : ℕ) (wolf_and_nobel : ℕ) :
  total = 50 →
  wolf = 31 →
  wolf_and_nobel = 14 →
  (total - wolf) = 2 * (total - wolf - 3) / 2 →
  ∃ (nobel : ℕ), nobel = 25 ∧ nobel = wolf_and_nobel + (total - wolf - 3) / 2 + 3 := by
  sorry

#check nobel_laureates_count

end nobel_laureates_count_l678_67867


namespace fraction_irreducible_l678_67870

theorem fraction_irreducible (n m : ℕ) : 
  Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 := by
  sorry

end fraction_irreducible_l678_67870


namespace sugar_percentage_approx_l678_67831

-- Define the initial solution volume
def initial_volume : ℝ := 500

-- Define the initial composition percentages
def water_percent : ℝ := 0.60
def cola_percent : ℝ := 0.08
def orange_percent : ℝ := 0.10
def lemon_percent : ℝ := 0.12

-- Define the added components
def added_sugar : ℝ := 4
def added_water : ℝ := 15
def added_cola : ℝ := 9
def added_orange : ℝ := 5
def added_lemon : ℝ := 7
def added_ice : ℝ := 8

-- Calculate the new total volume
def new_volume : ℝ := initial_volume + added_sugar + added_water + added_cola + added_orange + added_lemon + added_ice

-- Define the theorem
theorem sugar_percentage_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |added_sugar / new_volume - 0.0073| < ε :=
sorry

end sugar_percentage_approx_l678_67831


namespace gcd_of_specific_powers_of_two_l678_67895

theorem gcd_of_specific_powers_of_two : Nat.gcd (2^2048 - 1) (2^2035 - 1) = 8191 := by
  sorry

end gcd_of_specific_powers_of_two_l678_67895
